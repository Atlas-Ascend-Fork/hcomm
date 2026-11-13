/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ARRAY_HPP
#define CCU_ARRAY_HPP

#include <cstdint>
#include <new>
#include <vector>

#include "ccu_types.h"
#include "ccu_primitives_impl.h"
#include "ccu_utils.hpp"
#include "ccu_variable.hpp"
#include "ccu_event.hpp"
#include "ccu_buffer.hpp"

namespace hcomm {
namespace ccu {

    template <typename T>
    struct CcuArrayTraits;

    template <>
    struct CcuArrayTraits<Variable> {
        using Handle = CcuVariableHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockVariableAlloc(h, n); }
        static CcuResult CreateByAcquire(Handle acqHandle, uint32_t index, Handle* h)
        {
            return CcuVariableGetByIndex(acqHandle, index, h);
        }
        static void SetHandle(Variable& v, Handle h) { v.handle = h; }
    };

    template <>
    struct CcuArrayTraits<Event> {
        using Handle = CcuEventHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockEventAlloc(h, n); }
        static CcuResult CreateByAcquire(Handle acqHandle, uint32_t index, Handle* h)
        {
            return CcuEventGetByIndex(acqHandle, index, h);
        }
        static void SetHandle(Event& e, Handle h) { e.handle = h; }
    };

    template <>
    struct CcuArrayTraits<CcuBuffer> {
        using Handle = CcuBufferHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockBufferAlloc(h, n); }
        static void SetHandle(CcuBuffer& b, Handle h) { b.handle = h; }
    };

    template <typename T>
    class Array final {
    public:
        explicit Array(uint32_t count) : count_(count)
        {
            if (count == 0) {
                return;
            }
            using H = typename CcuArrayTraits<T>::Handle;
            std::vector<H> handles(count);
            elems_ = static_cast<T*>(::operator new(sizeof(T) * count));
            for (uint32_t i = 0; i < count; ++i) {
                ::new (static_cast<void*>(&elems_[i])) T(detail::NoAllocTag{});
            }
            auto ret = CcuArrayTraits<T>::BlockAlloc(handles.data(), count);
            if (ret != CcuResult::CCU_SUCCESS) {
                Release();
                throw ::hcomm::ccu::detail::CcuException(ret, "Array BlockAlloc: failed");
            }
            for (uint32_t i = 0; i < count; ++i) {
                CcuArrayTraits<T>::SetHandle(elems_[i], handles[i]);
            }
        }

        Array(typename CcuArrayTraits<T>::Handle acqHandle, uint32_t count) : count_(count)
        {
            if (count == 0) {
                return;
            }
            using H = typename CcuArrayTraits<T>::Handle;
            elems_ = static_cast<T*>(::operator new(sizeof(T) * count));
            for (uint32_t i = 0; i < count; ++i) {
                ::new (static_cast<void*>(&elems_[i])) T(detail::NoAllocTag{});
            }
            for (uint32_t i = 0; i < count; ++i) {
                H handle{};
                auto ret = CcuArrayTraits<T>::CreateByAcquire(acqHandle, i, &handle);
                if (ret != CcuResult::CCU_SUCCESS) {
                    std::string errMsg = "Array creation failed at index " + std::to_string(i)
                                         + ", requested count=" + std::to_string(count)
                                         + "; the acquire handle likely holds fewer resources than count";
                    Release();
                    throw ::hcomm::ccu::detail::CcuException(ret, errMsg.c_str());
                }
                CcuArrayTraits<T>::SetHandle(elems_[i], handle);
            }
        }

        ~Array() { Release(); }

        Array(const Array&) = delete;
        Array& operator=(const Array&) = delete;

        Array(Array&& other) noexcept : elems_(other.elems_), count_(other.count_)
        {
            other.elems_ = nullptr;
            other.count_ = 0;
        }

        Array& operator=(Array&& other) noexcept
        {
            if (this != &other) {
                // 只释放自身持有的资源，不可调用 this->~Array()：显式析构会结束本对象生命周期，
                // 之后再访问成员或返回 *this 属未定义行为
                Release();
                elems_ = other.elems_;
                count_ = other.count_;
                other.elems_ = nullptr;
                other.count_ = 0;
            }
            return *this;
        }

        T& operator[](uint32_t i) { return elems_[i]; }
        const T& operator[](uint32_t i) const { return elems_[i]; }
        T* data() { return elems_; }
        const T* data() const { return elems_; }
        uint32_t size() const { return count_; }

    private:
        // 销毁元素并归还存储，使对象回到合法的空状态；不结束对象自身生命周期，可被析构与移动赋值共用
        void Release() noexcept
        {
            if (elems_ == nullptr) {
                count_ = 0;
                return;
            }
            for (uint32_t i = 0; i < count_; ++i) {
                elems_[i].~T();
            }
            ::operator delete(elems_);
            elems_ = nullptr;
            count_ = 0;
        }

        T* elems_{nullptr};
        uint32_t count_{0};
    };

} // namespace ccu
} // namespace hcomm

#endif // CCU_ARRAY_HPP
